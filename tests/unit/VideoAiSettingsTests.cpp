// RTX Video SR model families (standard 1..4, high bitrate 16..19, streaming
// 21/23) with tiers, double SR and the supersample shrink across families, and
// persistence. Older readers reject the Video Effects values, so files that use
// them (as the base or the second SR) get a new version (settings presets) or a
// .video-ai sidecar (preset library, mode session); the older file stays as it was.
// Since 2026-11-08 double SR also takes DLSS SR and FSR in either pass; a DLSS SR /
// FSR second pass gets its own version again (v36, library v19 / session v14 in
// a .dual-sr-any sidecar).
#include "veyra/engine/GraphDescription.h"
#include "veyra/engine/PresetStore.h"
#include "veyra/engine/PresetLibrary.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
using namespace veyra::engine;
using veyra::pipeline::SrTarget;using veyra::pipeline::SrPrescale;using veyra::pipeline::SrOutput;
namespace {
unsigned checks=0,failures=0;
void check(bool ok,const char* why){++checks;if(!ok){++failures;std::cout<<"FAIL "<<why<<'\n';}}
std::string bytes(const std::filesystem::path& p){std::ifstream f(p,std::ios::binary);return {(std::istreambuf_iterator<char>(f)),{}};}
std::filesystem::path sidecar(const std::filesystem::path& p,const wchar_t* suffix=L".video-ai"){return std::filesystem::path(p).concat(suffix);}
// Every file of one library: the base name plus the sidecars older versions chained onto it.
std::map<std::wstring,std::string> files(const std::filesystem::path& base){
    std::map<std::wstring,std::string> out;
    for(const auto& e:std::filesystem::directory_iterator(base.parent_path()))
        if(e.path().filename().wstring().starts_with(base.filename().wstring()))out[e.path().filename().wstring()]=bytes(e.path());
    return out;
}
bool kept(const std::map<std::wstring,std::string>& before,const std::map<std::wstring,std::string>& after){
    return std::all_of(before.begin(),before.end(),[&](const auto& f){const auto it=after.find(f.first);return it!=after.end()&&it->second==f.second;});
}
veyra::pipeline::EnhanceGraphDesc describe(const EnhancementSettings& s,uint32_t width,uint32_t height,veyra::pipeline::ResolutionPlan* plan=nullptr,
                                           bool nvidia=true,bool still=false,bool amdAdapter=false){
    StageRequest request;request.sr=s.sr;request.width=width;request.height=height;request.nvidiaAdapter=nvidia;request.amdAdapter=amdAdapter;request.stillImage=still;
    veyra::pipeline::EnhanceGraphDesc desc;const auto p=describeStages(request,s,desc);
    if(plan)*plan=p;
    return desc;
}
}
int wmain(int argc,wchar_t** argv){
    if(argc!=2||!std::filesystem::path(argv[1]).is_absolute())return 2;
    const std::filesystem::path root=argv[1];if(!std::filesystem::create_directories(root))return 2;

    // Stored values: 0 DLSS, 5 FSR, and the three RTX Video SR families.
    for(uint32_t q=0;q<40;++q){
        const bool rtx=(q>=1&&q<=4)||(q>=16&&q<=19)||q==21||q==23;
        check(rtxVideoSr(q)==rtx,"RTX Video SR family values");
        check(validVideoSrQuality(q)==(rtx||q==0||q==kVideoSrFsr||spatialCnnVideoSr(q)),"stored SR quality table");
        EnhancementSettings s;s.sr=true;s.videoSrQuality=q;
        check(s.validate().empty()==validVideoSrQuality(q),"validation accepts exactly the stored SR qualities");
        check(ngxVideoSr(q)==(q>=1&&q<=4)&&vfxVideoSr(q)==(rtx&&q>4),"standard runs on NGX, the others on Video Effects");
    }
    check(vfxVideoSrNeedsAmpere(21)&&vfxVideoSrNeedsAmpere(23)&&!vfxVideoSrNeedsAmpere(16)&&!vfxVideoSrNeedsAmpere(19)&&!vfxVideoSrNeedsAmpere(4),
          "only the streaming family needs Ampere");
    // Family + tier <-> stored value.
    for(uint32_t tier=1;tier<=4;++tier){
        check(videoSrQualityFor(VideoSrFamily::Standard,tier)==tier,"standard tiers are 1..4");
        check(videoSrQualityFor(VideoSrFamily::HighBitrate,tier)==15+tier,"high bitrate tiers are 16..19");
        check(videoSrQualityFor(VideoSrFamily::Streaming,tier)==(tier<=2?21u:23u),"streaming has medium and ultra only");
        for(auto family:{VideoSrFamily::Standard,VideoSrFamily::HighBitrate}){
            const auto q=videoSrQualityFor(family,tier);
            check(videoSrFamily(q)==family&&videoSrTier(q)==tier,"family and tier round trip");
        }
    }
    check(videoSrFamily(21)==VideoSrFamily::Streaming&&videoSrTier(21)==2&&videoSrFamily(23)==VideoSrFamily::Streaming&&videoSrTier(23)==4,
          "streaming values report medium and ultra");
    check(videoSrQualityFor(VideoSrFamily::HighBitrate,0)==16&&videoSrQualityFor(VideoSrFamily::HighBitrate,9)==19,"tiers clamp to 1..4");

    // Double SR and the supersample shrink accept every family, also mixed.
    {
        EnhancementSettings s;s.sr=true;s.videoSrQuality=19;s.dualSrEnabled=true;s.dualSrQuality=23;s.srPrescale=SrPrescale::P1080;
        check(s.validate().empty(),"high bitrate base with a streaming second pass");
        auto active=s.activeVideoSettings();
        check(active.dualSrEnabled&&active.dualSrQuality==23&&active.srPrescale==SrPrescale::P1080,"double SR stays active on high bitrate");
        for(uint32_t bad2:{6u,15u,20u,22u,30u}){auto bad=s;bad.dualSrQuality=bad2;check(!bad.validate().empty(),"second SR must be a stored SR value");}
        for(uint32_t good2:{0u,kVideoSrFsr}){auto good=s;good.dualSrQuality=good2;check(good.validate().empty(),"DLSS SR and FSR are valid second passes");}
        s.videoSrQuality=4;s.dualSrQuality=17;check(s.validate().empty()&&s.activeVideoSettings().dualSrEnabled,"standard base with a high bitrate second pass");
        s.videoSrQuality=0;check(s.activeVideoSettings().dualSrEnabled,"a DLSS SR base runs double SR");
        s.videoSrQuality=kVideoSrFsr;check(s.activeVideoSettings().dualSrEnabled,"an FSR base runs double SR");
        // The DLSS SR model belongs to every DLSS SR pass, the second one too.
        auto model=s;model.videoSrQuality=4;model.dualSrQuality=0;model.dlssSrModel=1;
        check(model.activeVideoSettings().dlssSrModel==1,"a DLSS SR second pass keeps the DLSS SR model");
        model.dualSrQuality=17;check(model.activeVideoSettings().dlssSrModel==0,"without a DLSS SR pass the model is inactive");
        // DLSS SR / FSR passes read the SR motion source, RTX Video SR passes do not.
        auto motion=s;motion.videoSrQuality=4;motion.dualSrQuality=kVideoSrFsr;
        auto spatial=motion;spatial.dualSrQuality=17;
        check(motion.temporalSr()&&!spatial.temporalSr(),"a temporal second pass needs the SR motion source");
        EnhancementSettings ss;ss.sr=true;ss.videoSrQuality=16;ss.srOutput=SrOutput::Source;
        check(ss.activeVideoSettings().srOutput==SrOutput::Source,"supersample shrink stays on high bitrate");
        ss.videoSrQuality=0;check(ss.activeVideoSettings().srOutput==SrOutput::Target,"DLSS SR drops the supersample shrink");
    }
    // The graph plan: two passes with each family's quality, supersample sizes, and no same-size restore for Video Effects.
    {
        EnhancementSettings s;s.sr=true;s.videoSrQuality=19;s.srTarget=SrTarget::Uhd4K;
        s.dualSrEnabled=true;s.dualSrQuality=23;s.dualSrTarget=SrTarget::Uhd4K;s.srPrescale=SrPrescale::P1080;
        const auto desc=describe(s,1280,720);
        check(desc.enableSr&&desc.dualSr&&desc.videoSrQuality==19&&desc.dualSrQuality==23,"double SR plan carries both families");
        check(desc.firstSrWidth==3840&&desc.firstSrHeight==2160&&desc.srInputWidth==1920&&desc.srInputHeight==1080&&
              desc.srOutputWidth==3840&&desc.srOutputHeight==2160&&desc.workWidth==3840,"double SR sizes: 720p -> 4K -> 1080p -> 4K");
        s.videoSrQuality=4;s.dualSrQuality=16;
        check(describe(s,1280,720).dualSr,"standard base with high bitrate second pass plans two passes");
        // DLSS SR and FSR in either pass (2026-11-08).
        s.videoSrQuality=0;s.dualSrQuality=0;
        const auto dlss=describe(s,1280,720);
        check(dlss.dualSr&&dlss.videoSrQuality==0&&dlss.dualSrQuality==0&&dlss.firstSrWidth==3840&&dlss.srInputWidth==1920&&dlss.workWidth==3840,
              "DLSS SR -> DLSS SR plans 720p -> 4K -> 1080p -> 4K");
        s.dualSrQuality=kVideoSrFsr;check(describe(s,1280,720).dualSr,"DLSS SR base with an FSR second pass");
        s.videoSrQuality=kVideoSrFsr;s.dualSrQuality=19;check(describe(s,1280,720).dualSr,"FSR base with a high bitrate second pass");
        s.videoSrQuality=17;s.dualSrQuality=0;check(describe(s,1280,720).dualSr,"high bitrate base with a DLSS SR second pass");
        // Adapters: FSR runs everywhere, the others on NVIDIA only.
        s.videoSrQuality=kVideoSrFsr;s.dualSrQuality=kVideoSrFsr;
        const auto amd=describe(s,1280,720,nullptr,false);
        check(amd.enableSr&&amd.dualSr&&amd.srInputWidth==1920,"FSR -> FSR double SR runs on a non-NVIDIA adapter");
        s.dualSrQuality=2;const auto amdSingle=describe(s,1280,720,nullptr,false);
        check(!amdSingle.dualSr&&amdSingle.enableSr,"a non-NVIDIA adapter keeps the single FSR pass when the second pass needs NVIDIA");
        // FSR skips still images, so a double SR with an FSR pass keeps one pass there.
        s.dualSrQuality=kVideoSrFsr;check(!describe(s,1280,720,nullptr,true,true).dualSr,"no double SR with FSR on a still image");
        s.videoSrQuality=0;s.dualSrQuality=0;check(describe(s,1280,720,nullptr,true,true).dualSr,"DLSS SR double SR on a still image");
        EnhancementSettings ss;ss.sr=true;ss.videoSrQuality=17;ss.srTarget=SrTarget::Uhd4K;ss.srOutput=SrOutput::Qhd;
        const auto sd=describe(ss,1280,720);
        check(sd.enableSr&&!sd.dualSr&&sd.srOutputWidth==3840&&sd.workWidth==2560&&sd.workHeight==1440,"high bitrate supersample: SR to 4K, shrink to 1440p");
        EnhancementSettings same;same.sr=true;same.srTarget=SrTarget::Uhd4K;same.videoSrQuality=19;
        check(!describe(same,3840,2160).enableSr,"Video Effects families do not run at the target size");
        same.videoSrQuality=4;check(describe(same,3840,2160).enableSr,"standard family restores at the target size");
    }

    EnhancementSettings vsr;vsr.sr=true;vsr.videoSrQuality=23;vsr.srTarget=SrTarget::Qhd;
    EnhancementSettings second;second.sr=true;second.videoSrQuality=4;second.dualSrEnabled=true;second.dualSrQuality=18;second.srPrescale=SrPrescale::P900;
    EnhancementSettings both=vsr;both.videoSrQuality=19;both.dualSrEnabled=true;both.dualSrQuality=21;both.srPrescale=SrPrescale::P1080;
    // DLSS SR / FSR second passes (2026-11-08).
    EnhancementSettings dlssSecond;dlssSecond.sr=true;dlssSecond.videoSrQuality=4;dlssSecond.dualSrEnabled=true;dlssSecond.dualSrQuality=0;
    dlssSecond.dlssSrModel=1;dlssSecond.srPrescale=SrPrescale::P1080;
    EnhancementSettings fsrSecond;fsrSecond.sr=true;fsrSecond.videoSrQuality=0;fsrSecond.dualSrEnabled=true;fsrSecond.dualSrQuality=kVideoSrFsr;
    fsrSecond.srPrescale=SrPrescale::P900;fsrSecond.srTarget=SrTarget::Uhd8K;

    // Settings presets: v35 only when a Video Effects value is stored.
    {
        PresetStore plain(root/L"plain.txt");check(plain.load()&&plain.put(L"plain",EnhancementSettings{}),"plain settings preset");
        check(!bytes(root/L"plain.txt").starts_with("VEYRA_PRESETS 35"),"settings without new values keep the older version");
        EnhancementSettings stdDual;stdDual.sr=true;stdDual.dualSrEnabled=true;stdDual.dualSrQuality=2;stdDual.srPrescale=SrPrescale::P1080;
        PresetStore standardDual(root/L"standard-dual.txt");
        check(standardDual.load()&&standardDual.put(L"std",stdDual)&&bytes(root/L"standard-dual.txt").starts_with("VEYRA_PRESETS 34"),"standard double SR keeps v34");
        PresetStore store(root/L"video-ai.txt");
        check(store.load()&&store.put(L"vsr",vsr)&&store.put(L"second",second)&&store.put(L"both",both),"settings presets with new values");
        check(bytes(root/L"video-ai.txt").starts_with("VEYRA_PRESETS 35"),"a Video Effects value writes v35");
        PresetStore reader(root/L"video-ai.txt");check(reader.load()&&reader.entries().size()==3,"v35 reload");
        const auto find=[&](const wchar_t* name)->const EnhancementSettings*{
            for(const auto& e:reader.entries())if(e.name==name)return &e.settings;return nullptr;};
        const auto* a=find(L"vsr");const auto* b=find(L"second");const auto* c=find(L"both");
        check(a&&a->sr&&a->videoSrQuality==23&&a->srTarget==SrTarget::Qhd,"v35 restores the base family and tier");
        check(b&&b->videoSrQuality==4&&b->dualSrEnabled&&b->dualSrQuality==18&&b->srPrescale==SrPrescale::P900,"v35 restores a Video Effects second pass");
        check(c&&c->videoSrQuality==19&&c->dualSrQuality==21&&c->srPrescale==SrPrescale::P1080,"v35 restores both passes");
        // A DLSS SR / FSR second pass: v36, older (v35) readers reject the file whole.
        PresetStore anySr(root/L"any-sr.txt");
        check(anySr.load()&&anySr.put(L"dlss",dlssSecond)&&anySr.put(L"fsr",fsrSecond),"settings presets with DLSS SR / FSR second passes");
        check(bytes(root/L"any-sr.txt").starts_with("VEYRA_PRESETS 36"),"a DLSS SR / FSR second pass writes v36");
        PresetStore anyReader(root/L"any-sr.txt");check(anyReader.load()&&anyReader.entries().size()==2,"v36 reload");
        for(const auto& e:anyReader.entries()){
            const auto& want=e.name==L"dlss"?dlssSecond:fsrSecond;
            check(e.settings.videoSrQuality==want.videoSrQuality&&e.settings.dualSrEnabled&&e.settings.dualSrQuality==want.dualSrQuality&&
                  e.settings.srPrescale==want.srPrescale&&e.settings.dlssSrModel==want.dlssSrModel&&e.settings.srTarget==want.srTarget,"v36 restores both passes");
        }
        PresetStore mixed(root/L"video-ai.txt");check(mixed.load()&&mixed.put(L"dlss",dlssSecond)&&bytes(root/L"video-ai.txt").starts_with("VEYRA_PRESETS 36"),
              "adding a DLSS SR second pass upgrades a v35 store to v36");
        PresetStore mixedReader(root/L"video-ai.txt");check(mixedReader.load()&&mixedReader.entries().size()==4,"v36 keeps the Video Effects entries");
    }

    // Preset library: new entries go to a v18 .video-ai sidecar; the older files stay.
    {
        const auto path=root/L"library.preset";
        {
            PresetLibrary old(path);old.setIncludeBuiltins(false);
            auto e=std::make_unique<PresetEntry>();e->name=L"old";EnhancementSettings s;e->chain=toChain(s);e->globals=ChainGlobalSettings::capture(s);
            check(old.load()&&old.put(*e),"older library entry");
        }
        const auto before=files(path);
        check(before.size()==1&&!before.begin()->second.starts_with("VEYRA_PRESET_LIBRARY 18")&&!before.begin()->first.ends_with(L".video-ai"),
              "library without new values keeps an older version and file");
        std::vector<std::unique_ptr<PresetEntry>> saved;
        for(const auto& [name,settings]:{std::pair{L"vsr",&vsr},std::pair{L"second",&second},std::pair{L"both",&both}}){
            PresetLibrary library(path);library.setIncludeBuiltins(false);check(library.load(),"library reload before save");
            auto e=std::make_unique<PresetEntry>();e->name=name;e->chain=toChain(*settings);e->globals=ChainGlobalSettings::capture(*settings);
            check(library.put(*e),"save an entry with a Video Effects value");saved.push_back(std::move(e));
        }
        const auto after=files(path);
        check(kept(before,after),"older library file untouched");
        check(after.size()==before.size()+1&&std::any_of(after.begin(),after.end(),[&](const auto& f){
            return !before.contains(f.first)&&f.first.ends_with(L".video-ai")&&f.second.starts_with("VEYRA_PRESET_LIBRARY 18");}),
              "library writes v18 to one new .video-ai sidecar");
        PresetLibrary reloaded(path);reloaded.setIncludeBuiltins(false);
        check(reloaded.load()&&reloaded.entries().size()==4,"library reload reads the sidecar");
        for(const auto& e:saved){
            const auto it=std::find_if(reloaded.entries().begin(),reloaded.entries().end(),[&](const auto& r){return r.name==e->name;});
            check(it!=reloaded.entries().end()&&*it==*e,"library restores the full entry");
            if(it==reloaded.entries().end()||!e->globals)continue;
            EnhancementSettings applied;PresetLibrary::apply(*it,applied);
            check(applied.videoSrQuality==e->globals->videoSrQuality&&applied.dualSrQuality==e->globals->dualSrQuality&&
                  applied.dualSrEnabled==e->globals->dualSrEnabled,"applying restores both SR passes");
        }
        // Removing the new values keeps the newer build on the sidecar.
        PresetLibrary trimmed(path);trimmed.setIncludeBuiltins(false);check(trimmed.load(),"reload before removal");
        for(const auto* name:{L"vsr",L"second",L"both"}){
            const auto& list=trimmed.entries();
            const auto it=std::find_if(list.begin(),list.end(),[&](const auto& r){return r.name==name;});
            check(it!=list.end()&&trimmed.erase(size_t(it-list.begin())),"remove new entry");
        }
        const auto removed=files(path);
        check(kept(before,removed)&&removed.size()==before.size()+1,"removal writes the sidecar, not the older file");
    }

    // Preset library with DLSS SR / FSR second passes: v19 in a .dual-sr-any sidecar.
    {
        const auto path=root/L"any-sr.preset";
        {
            PresetLibrary old(path);old.setIncludeBuiltins(false);
            auto e=std::make_unique<PresetEntry>();e->name=L"old";e->chain=toChain(vsr);e->globals=ChainGlobalSettings::capture(vsr);
            check(old.load()&&old.put(*e),"library entry with a Video Effects value");
        }
        const auto before=files(path);
        check(before.size()==1&&before.begin()->first.ends_with(L".video-ai")&&before.begin()->second.starts_with("VEYRA_PRESET_LIBRARY 18"),
              "a Video Effects library stays v18 in its sidecar");
        std::vector<std::unique_ptr<PresetEntry>> saved;
        for(const auto& [name,settings]:{std::pair{L"dlss",&dlssSecond},std::pair{L"fsr",&fsrSecond}}){
            PresetLibrary library(path);library.setIncludeBuiltins(false);check(library.load(),"library reload before save");
            auto e=std::make_unique<PresetEntry>();e->name=name;e->chain=toChain(*settings);e->globals=ChainGlobalSettings::capture(*settings);
            check(library.put(*e),"save an entry with a DLSS SR / FSR second pass");saved.push_back(std::move(e));
        }
        const auto after=files(path);
        check(kept(before,after),"older library files untouched");
        check(after.size()==before.size()+1&&std::any_of(after.begin(),after.end(),[&](const auto& f){
            return !before.contains(f.first)&&f.first.ends_with(L".dual-sr-any")&&f.second.starts_with("VEYRA_PRESET_LIBRARY 19");}),
              "library writes v19 to one new .dual-sr-any sidecar");
        PresetLibrary reloaded(path);reloaded.setIncludeBuiltins(false);
        check(reloaded.load()&&reloaded.entries().size()==3,"library reload reads the newest sidecar");
        for(const auto& e:saved){
            const auto it=std::find_if(reloaded.entries().begin(),reloaded.entries().end(),[&](const auto& r){return r.name==e->name;});
            check(it!=reloaded.entries().end()&&*it==*e,"library restores the full DLSS SR / FSR entry");
            if(it==reloaded.entries().end()||!e->globals)continue;
            EnhancementSettings applied;PresetLibrary::apply(*it,applied);
            check(applied.dualSrEnabled&&applied.dualSrQuality==e->globals->dualSrQuality&&applied.videoSrQuality==e->globals->videoSrQuality&&
                  applied.validate().empty(),"applying restores a DLSS SR / FSR second pass");
        }
    }

    // Mode session (node mode, high bitrate base with a streaming second pass).
    {
        auto session=std::make_unique<ChainSession>(ChainSession::initial(both));check(session->select(ChainMode::Node),"node mode initializes");
        auto editor=std::make_shared<NodeEditorDocument>();editor->nodes=session->configurations[1].chain;
        editor->globals=ChainGlobalSettings::capture(both);check(editor->layout.initialize(editor->nodes).accepted,"node draft layout");
        session->configurations[1].editor=editor;check(session->valid(),"session valid");
        const auto path=root/L"modes.session";
        ChainSessionStore writer(path);check(writer.save(*session),"session save");
        check(!std::filesystem::exists(path)&&bytes(sidecar(path)).starts_with("VEYRA_CHAIN_SESSION 13"),"session writes v13 to the .video-ai sidecar only");
        auto restored=std::make_unique<ChainSession>();ChainSessionStore reader(path);
        check(reader.load(*restored)&&*session==*restored,"session restart restores both SR passes");
        auto plain=std::make_unique<ChainSession>(ChainSession::initial(EnhancementSettings{}));
        const auto plainPath=root/L"plain.session";ChainSessionStore plainWriter(plainPath);
        check(plainWriter.save(*plain)&&std::filesystem::exists(plainPath)&&!std::filesystem::exists(sidecar(plainPath)),"plain session keeps the older file");
        // A DLSS SR second pass: v14 in the .dual-sr-any sidecar, nothing else written.
        auto anySession=std::make_unique<ChainSession>(ChainSession::initial(dlssSecond));
        const auto anyPath=root/L"any-sr.session";ChainSessionStore anyWriter(anyPath);
        check(anyWriter.save(*anySession)&&!std::filesystem::exists(anyPath)&&!std::filesystem::exists(sidecar(anyPath))&&
              bytes(sidecar(anyPath,L".dual-sr-any")).starts_with("VEYRA_CHAIN_SESSION 14"),"session writes v14 to the .dual-sr-any sidecar only");
        auto anyRestored=std::make_unique<ChainSession>();ChainSessionStore anyReader(anyPath);
        check(anyReader.load(*anyRestored)&&*anySession==*anyRestored,"session restart restores the DLSS SR second pass");
    }

    // Milestone 3: FSR SR & AMF FRC settings normalization, validation, graph description, and rebuild checks.
    {
        EnhancementSettings fsr;
        check(fsr.fsrSrVersionMode==0&&fsr.fsrSrReactiveMask&&!fsr.fsrSrSharpening&&fsr.fsrSrSharpness==0.5f,"FSR SR defaults preserved");
        check(fsr.amfFrcProfile==0&&fsr.amfFrcMvSearch==0&&!fsr.amfFrcUseFutureFrame,"AMF FRC defaults preserved");

        fsr.sr=true;fsr.videoSrQuality=kVideoSrFsr;
        fsr.fsrSrVersionMode=2;fsr.fsrSrReactiveMask=false;fsr.fsrSrSharpening=true;fsr.fsrSrSharpness=0.8f;
        check(fsr.validate().empty(),"valid FSR SR settings accepted");
        const auto activeFsr=fsr.activeVideoSettings();
        check(activeFsr.fsrSrVersionMode==2&&!activeFsr.fsrSrReactiveMask&&activeFsr.fsrSrSharpening&&activeFsr.fsrSrSharpness==0.8f,"activeVideoSettings preserves FSR SR config when FSR active");

        auto fsrSecond=fsr;fsrSecond.videoSrQuality=4;fsrSecond.dualSrEnabled=true;fsrSecond.dualSrQuality=kVideoSrFsr;
        check(fsrSecond.activeVideoSettings().fsrSrVersionMode==2,"activeVideoSettings preserves FSR SR config on dual SR second pass");

        auto nonFsr=fsr;nonFsr.videoSrQuality=0;
        const auto normNonFsr=nonFsr.activeVideoSettings();
        check(normNonFsr.fsrSrVersionMode==0&&normNonFsr.fsrSrReactiveMask&&!normNonFsr.fsrSrSharpening&&normNonFsr.fsrSrSharpness==0.5f,"activeVideoSettings normalizes FSR SR fields when FSR inactive");

        auto bad=fsr;bad.fsrSrVersionMode=kFsrSrVersionModeCount;check(!bad.validate().empty(),"reject out-of-range fsrSrVersionMode");
        bad=fsr;bad.fsrSrSharpness=1.5f;check(!bad.validate().empty(),"reject out-of-range fsrSrSharpness");

        const auto fsrDesc=describe(fsr,1920,1080);
        check(fsrDesc.fsrSrVersionMode==2&&!fsrDesc.fsrSrReactiveMask&&fsrDesc.fsrSrSharpening&&fsrDesc.fsrSrSharpness==0.8f,"describeStages propagates FSR SR config");

        auto fsrNext=fsr;fsrNext.fsrSrVersionMode=1;
        check(requiresGraphRebuild(fsr,fsrNext),"changing fsrSrVersionMode requires rebuild");
        fsrNext=fsr;fsrNext.fsrSrReactiveMask=true;
        check(requiresGraphRebuild(fsr,fsrNext),"changing fsrSrReactiveMask requires rebuild");
        fsrNext=fsr;fsrNext.fsrSrSharpness=0.3f;
        check(!requiresGraphRebuild(fsr,fsrNext),"changing fsrSrSharpness updates live without rebuild");

        EnhancementSettings amf;amf.multiplier=2;amf.frameGenerationBackend=FrameGenerationBackend::AmdFrc;
        amf.amfFrcProfile=3;amf.amfFrcMvSearch=2;amf.amfFrcUseFutureFrame=true;
        check(amf.validate().empty(),"valid AMF FRC settings accepted");
        const auto activeAmf=amf.activeVideoSettings();
        check(activeAmf.amfFrcProfile==3&&activeAmf.amfFrcMvSearch==2&&activeAmf.amfFrcUseFutureFrame,"activeVideoSettings preserves AMF FRC config");
        auto nonAmf=amf;nonAmf.frameGenerationBackend=FrameGenerationBackend::Dlss;
        const auto normNonAmf=nonAmf.activeVideoSettings();
        check(normNonAmf.amfFrcProfile==0&&normNonAmf.amfFrcMvSearch==0&&!normNonAmf.amfFrcUseFutureFrame,"activeVideoSettings normalizes AMF FRC fields when inactive");

        bad=amf;bad.amfFrcProfile=kAmfFrcProfileCount;check(!bad.validate().empty(),"reject out-of-range amfFrcProfile");
        bad=amf;bad.amfFrcMvSearch=kAmfFrcMvSearchCount;check(!bad.validate().empty(),"reject out-of-range amfFrcMvSearch");

        const auto amfDesc=describe(amf,1920,1080);
        check(amfDesc.amfFrcProfile==3&&amfDesc.amfFrcMvSearch==2&&amfDesc.amfFrcUseFutureFrame,"describeStages propagates AMF FRC config");
        auto amfNext=amf;amfNext.amfFrcUseFutureFrame=false;
        check(requiresGraphRebuild(amf,amfNext),"changing AMF FRC config requires rebuild");
    }

    // M1-M3 persistence (2026-11-24): the seven FSR SR / AMF FRC options are a
    // new settings version (v38) because an older reader would take them as the
    // next entry's fields. The switches they belong to (sr + FSR base, AMF FRC
    // frame generation) are stored in the same file, so a reload has to restore
    // both. Files that never touch the options keep their own version and bytes.
    {
        EnhancementSettings ai;ai.sr=true;ai.videoSrQuality=kVideoSrFsr;
        ai.fsrSrVersionMode=2;ai.fsrSrReactiveMask=false;ai.fsrSrSharpening=true;ai.fsrSrSharpness=0.8f;
        ai.multiplier=2;ai.frameGenerationBackend=FrameGenerationBackend::AmdFrc;
        ai.amfFrcProfile=3;ai.amfFrcMvSearch=2;ai.amfFrcUseFutureFrame=true;
        check(ai.validate().empty(),"M1-M3 preset values validate");
        const auto aiActive=ai.activeVideoSettings();
        check(aiActive.fsrSrVersionMode==2&&!aiActive.fsrSrReactiveMask&&aiActive.fsrSrSharpening&&aiActive.fsrSrSharpness==0.8f&&
              aiActive.amfFrcProfile==3&&aiActive.amfFrcMvSearch==2&&aiActive.amfFrcUseFutureFrame,
              "activeVideoSettings keeps the M1-M3 values while FSR SR and AMF FRC run");
        const auto aiPath=root/L"m1-m3.txt";
        PresetStore writer(aiPath);
        check(writer.load()&&writer.put(L"ai",ai),"settings preset with M1-M3 options");
        check(bytes(aiPath).starts_with("VEYRA_PRESETS 38"),"M1-M3 options write v38");
        PresetStore reader(aiPath);
        const bool reloaded=reader.load()&&reader.entries().size()==1;
        check(reloaded,"v38 reload");
        if(reloaded){
            const auto& got=reader.entries().front().settings;
            check(got.fsrSrVersionMode==2&&!got.fsrSrReactiveMask&&got.fsrSrSharpening&&got.fsrSrSharpness==0.8f&&
                  got.amfFrcProfile==3&&got.amfFrcMvSearch==2&&got.amfFrcUseFutureFrame,"v38 restores all seven M1-M3 options");
            check(got.sr&&got.videoSrQuality==kVideoSrFsr&&got.multiplier==2&&got.frameGenerationBackend==FrameGenerationBackend::AmdFrc,
                  "v38 restores the FSR SR / AMF FRC settings the options belong to");
            check(got.validate().empty(),"a reloaded v38 preset is still valid");
        }
        const auto v38=bytes(aiPath);
        PresetStore resave(aiPath);
        const bool same38=resave.load()&&resave.entries().size()==1&&
            resave.put(resave.entries().front().name,resave.entries().front().settings,true);
        check(same38&&bytes(aiPath)==v38,"re-saving a v38 file reproduces the same bytes");
        // A store that never touches the seven options keeps its older version.
        EnhancementSettings protect;protect.exclusion.enabled=true;
        check(protect.validate().empty(),"v37 exclusion settings validate");
        const auto v37Path=root/L"m1-m3-v37.txt";
        PresetStore v37writer(v37Path);
        check(v37writer.load()&&v37writer.put(L"protect",protect),"settings preset with an exclusion mask");
        check(bytes(v37Path).starts_with("VEYRA_PRESETS 37"),"M1-M3 defaults keep a v37 store at v37");
        const auto v37bytes=bytes(v37Path);
        PresetStore v37resave(v37Path);
        const bool same37=v37resave.load()&&v37resave.entries().size()==1&&
            v37resave.put(v37resave.entries().front().name,v37resave.entries().front().settings,true);
        check(same37&&bytes(v37Path)==v37bytes,"re-saving a v37 file reproduces the same bytes");
        PresetStore upgrade(v37Path);check(upgrade.load(),"v37 reload before upgrade");
        check(upgrade.put(L"ai",ai)&&bytes(v37Path).starts_with("VEYRA_PRESETS 38"),"adding an M1-M3 entry upgrades a v37 store to v38");
        PresetStore upgradeReader(v37Path);
        const bool upgraded=upgradeReader.load()&&upgradeReader.entries().size()==2;
        check(upgraded,"v38 keeps the older v37 entry");
        if(upgraded){
            const auto& kept37=upgradeReader.entries().front().settings;
            check(kept37.exclusion.enabled&&kept37.fsrSrVersionMode==0&&kept37.fsrSrReactiveMask&&!kept37.fsrSrSharpening&&
                  kept37.fsrSrSharpness==0.5f&&kept37.amfFrcProfile==0&&kept37.amfFrcMvSearch==0&&!kept37.amfFrcUseFutureFrame,
                  "the v38 layout gives an older entry the M1-M3 defaults");
        }
        // A newer build's file is rejected whole and never overwritten.
        const auto futurePath=root/L"m1-m3-future.txt";
        const std::string future="VEYRA_PRESETS 40\n\"\" 1\n\"new\" 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 0 0 0 1 2\n";
        {std::ofstream f(futurePath,std::ios::binary);f<<future;}
        PresetStore futureReader(futurePath);
        check(!futureReader.load(),"a newer settings version is rejected");
        check(!futureReader.error().empty(),"rejecting a newer version reports why");
        check(bytes(futurePath)==future&&!futureReader.put(L"ai",ai),"a newer settings file is never overwritten");
        // A truncated v38 line (the options cut off) is rejected, not half-read.
        const auto cut=v38.substr(0,v38.size()>14?v38.size()-14:0);
        const auto cutPath=root/L"m1-m3-cut.txt";
        {std::ofstream f(cutPath,std::ios::binary);f<<cut;}
        PresetStore cutReader(cutPath);
        check(!cutReader.load(),"a truncated v38 line is rejected");
        check(bytes(cutPath)==cut,"a truncated v38 file is left alone");
    }

    // Spatial single-frame SR (W6, 2026-11-24): ArtCNN 4-bit/16-bit (24/25), AMF
    // HQScaler (26), AMF VQ Enhancer (27) and ORT-DML SPAN / RealPLKSR (28/29).
    // They are vendor neutral and never temporal, so they need no motion vectors
    // and run on AMD/Intel adapters too. Retired ID 6 stays invalid.
    {
        const uint32_t spatial[]{24u,25u,26u,27u,28u,29u};
        for(uint32_t q:spatial)check(validVideoSrQuality(q),"spatial SR qualities are stored values");
        for(uint32_t q:{6u,7u,20u,22u,30u})check(!validVideoSrQuality(q),"retired or unassigned SR qualities stay invalid");
        for(uint32_t q:spatial){
            check(!temporalVideoSr(q),"spatial SR never reads the SR motion source");
            check(!rtxVideoSr(q),"spatial SR is not an RTX Video SR family value");
            check(spatialCnnVideoSr(q),"the spatial family predicate covers 24..29");
            EnhancementSettings s;s.sr=true;s.videoSrQuality=q;
            check(s.validate().empty()&&!s.temporalSr(),"spatial SR validates and is spatial");
        }
        // Disjoint partition: ArtCNN {24,25}, AMF HQ {26}, AMF VQ {27}, DML {28,29}.
        for(uint32_t q:spatial){
            check(artcnnVideoSr(q)==(q==24||q==25),"ArtCNN covers 24/25 only");
            check(amfHqVideoSr(q)==(q==26),"AMF HQScaler is 26 only");
            check(amfVqVideoSr(q)==(q==27),"AMF VQ Enhancer is 27 only");
            check(amfVideoSr(q)==(q==26||q==27),"AMF covers 26/27 only");
            check(dmlVideoSr(q)==(q==28||q==29),"ORT-DML covers 28/29 only");
            const int families=int(artcnnVideoSr(q))+int(amfVideoSr(q))+int(dmlVideoSr(q));
            check(families==1,"each spatial SR value belongs to exactly one family");
        }
        for(uint32_t q:{0u,kVideoSrFsr,1u,2u,3u,4u,16u,19u,21u,23u,6u,30u}){
            check(!spatialCnnVideoSr(q),"non-spatial SR values are outside spatialCnnVideoSr");
            check(!artcnnVideoSr(q)&&!amfVideoSr(q)&&!dmlVideoSr(q),"non-spatial SR values are outside every spatial family");
        }
        // A spatial base or second pass never asks for the SR motion source.
        EnhancementSettings dual;dual.sr=true;dual.videoSrQuality=27;dual.dualSrEnabled=true;dual.dualSrQuality=27;dual.srPrescale=SrPrescale::P1080;
        check(dual.validate().empty(),"a spatial second pass validates");
        check(!dual.temporalSr(),"spatial SR never needs the SR motion source, in either pass");
        dual.videoSrQuality=26;dual.dualSrQuality=29;
        check(!dual.temporalSr(),"mixed spatial passes stay motion-free");
        dual.dualSrQuality=0;
        check(dual.temporalSr(),"a DLSS SR second pass still reads the SR motion source");
    }

    // Graph description by adapter vendor: vendor-neutral spatial IDs (ArtCNN
    // 24/25, DirectML 28/29) enable SR on NVIDIA, AMD and Intel when enlarging;
    // AMD AMF video SR (26/27) enables SR on AMD adapters only.
    {
        EnhancementSettings s;s.sr=true;s.srTarget=SrTarget::Uhd4K;
        for(uint32_t q:{24u,25u,28u,29u}){
            s.videoSrQuality=q;
            const auto descIntel=describe(s,1920,1080,nullptr,false,false,false);
            const auto descAmd=describe(s,1920,1080,nullptr,false,false,true);
            check(descIntel.enableSr&&descIntel.videoSrQuality==q,"vendor-neutral spatial SR enlarges 1080p -> 4K on Intel");
            check(descAmd.enableSr&&descAmd.videoSrQuality==q,"vendor-neutral spatial SR enlarges 1080p -> 4K on AMD");
        }
        for(uint32_t q:{26u,27u}){
            s.videoSrQuality=q;
            const auto descAmd=describe(s,1920,1080,nullptr,false,false,true);
            check(descAmd.enableSr&&descAmd.videoSrQuality==q,"AMF video SR enlarges 1080p -> 4K on AMD");
            check(!describe(s,1920,1080,nullptr,false,false,false).enableSr,"AMF video SR stays off on Intel");
            check(!describe(s,1920,1080,nullptr,true,false,false).enableSr,"AMF video SR stays off on NVIDIA");
        }
        // Already at the target: only the 1x AMF VQ Enhancer (27) runs, and only on AMD.
        s.videoSrQuality=27;
        check(describe(s,3840,2160,nullptr,false,false,true).enableSr,"AMF VQ Enhancer runs 1x on AMD");
        check(!describe(s,3840,2160,nullptr,false,false,false).enableSr,"AMF VQ Enhancer stays off at 1x on Intel");
        check(!describe(s,3840,2160,nullptr,true,false,false).enableSr,"AMF VQ Enhancer stays off at 1x on NVIDIA");
        for(uint32_t q:{24u,25u,26u,28u,29u}){
            s.videoSrQuality=q;
            check(!describe(s,3840,2160,nullptr,false,false,true).enableSr,"the other spatial methods never run at the target size on AMD");
            check(!describe(s,3840,2160,nullptr,true,false,false).enableSr,"the other spatial methods never run at the target size on NVIDIA");
        }
        s.videoSrQuality=27;
        check(!describe(s,3840,2160,nullptr,false,true,true).enableSr,"AMF VQ Enhancer skips still images on AMD");
        check(!describe(s,3840,2160,nullptr,true,true,false).enableSr,"AMF VQ Enhancer skips still images on NVIDIA");
        // The vendor gate still rejects the RTX families off NVIDIA.
        s.videoSrQuality=4;
        check(!describe(s,1920,1080,nullptr,false,false,true).enableSr,"standard RTX Video SR stays NVIDIA only");
    }

    // Settings persistence: spatial SR is a v39 layout identical to v38, so an
    // older reader rejects the header instead of loading an SR it cannot run.
    {
        const auto spatialPath=root/L"spatial-sr.txt";
        std::vector<EnhancementSettings> saved;
        for(uint32_t q:{24u,25u,26u,27u,28u,29u}){
            EnhancementSettings s;s.sr=true;s.srTarget=SrTarget::Uhd4K;s.videoSrQuality=q;
            PresetStore store(spatialPath);
            const std::wstring name=L"spatial"+std::to_wstring(q);
            check(store.load()&&store.put(name,s),"settings preset with a spatial SR value");
            saved.push_back(s);
        }
        check(bytes(spatialPath).starts_with("VEYRA_PRESETS 39"),"spatial SR writes v39");
        PresetStore reader(spatialPath);
        const bool reloaded=reader.load()&&reader.entries().size()==6;
        check(reloaded,"v39 reload");
        if(reloaded){
            for(size_t i=0;i<6;++i){
                const auto& got=reader.entries()[i].settings;
                check(got.sr&&got.srTarget==SrTarget::Uhd4K&&got.videoSrQuality==saved[i].videoSrQuality,"v39 restores the spatial SR quality exactly");
            }
        }
        const auto v39=bytes(spatialPath);
        PresetStore resave(spatialPath);
        const bool same39=resave.load()&&resave.entries().size()==6&&
            resave.put(resave.entries().front().name,resave.entries().front().settings,true);
        check(same39&&bytes(spatialPath)==v39,"re-saving a v39 file reproduces the same bytes");
        // A store without spatial SR keeps v38; the spatial block is not emitted.
        EnhancementSettings ai;ai.sr=true;ai.videoSrQuality=kVideoSrFsr;ai.fsrSrVersionMode=2;ai.fsrSrSharpness=0.8f;
        const auto v38Path=root/L"spatial-v38.txt";
        PresetStore v38writer(v38Path);check(v38writer.load()&&v38writer.put(L"ai",ai),"settings preset without spatial SR");
        check(bytes(v38Path).starts_with("VEYRA_PRESETS 38"),"no spatial SR keeps a v38 store at v38");
        // Adding a spatial entry upgrades the same file to v39 and keeps v38 defaults.
        PresetStore upgrade(v38Path);check(upgrade.load(),"v38 reload before upgrade");
        EnhancementSettings spatial;spatial.sr=true;spatial.videoSrQuality=26;
        check(upgrade.put(L"hq",spatial)&&bytes(v38Path).starts_with("VEYRA_PRESETS 39"),"adding a spatial entry upgrades to v39");
        PresetStore upgraded(v38Path);
        const bool ok=upgraded.load()&&upgraded.entries().size()==2;
        check(ok,"v39 keeps the older v38 entry");
        if(ok){
            const auto& kept38=upgraded.entries().front().settings;
            check(kept38.videoSrQuality==kVideoSrFsr&&kept38.fsrSrVersionMode==2&&kept38.fsrSrSharpness==0.8f,
                  "the v39 layout restores older v38 fields");
            const auto& gotSpatial=upgraded.entries().back().settings;
            check(gotSpatial.videoSrQuality==26&&gotSpatial.fsrSrVersionMode==0&&gotSpatial.fsrSrReactiveMask&&
                  !gotSpatial.fsrSrSharpening&&gotSpatial.fsrSrSharpness==0.5f,"a spatial entry gets the v38 option defaults");
        }
        // A v40 file is a newer layout: rejected whole, never overwritten.
        const auto v40Path=root/L"spatial-v40.txt";
        const std::string v40="VEYRA_PRESETS 40\n\"\" 1\n\"new\" 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 0 0 0 1 2\n";
        {std::ofstream f(v40Path,std::ios::binary);f<<v40;}
        PresetStore v40reader(v40Path);
        check(!v40reader.load(),"a v40 settings version is rejected");
        check(bytes(v40Path)==v40&&!v40reader.put(L"hq",spatial),"a v40 settings file is never overwritten");
    }

    std::cout<<"Video AI settings "<<checks<<" checks, "<<failures<<" failures\n";return failures?1:0;
}
